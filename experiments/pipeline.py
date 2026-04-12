"""
pipeline.py -- 전체 A/B 테스트 오케스트레이션

생성 → Rule 평가 → LLM 평가 → 집계 → 시각화까지 순차 실행한다.

사용법:
    python -m experiments.pipeline                    # 전체 파이프라인
    python -m experiments.pipeline --stage eval       # 평가만
    python -m experiments.pipeline --stage report     # 리포트만
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from experiments.config import (
    ARTIFACTS_DIR,
    EVALS_DIR,
    LLM_EVAL_CRITERIA,
    LOGS_DIR,
    MODELS,
    WIN_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── 1단계: 생성 ─────────────────────────────────────────────────────────────

def stage_generate(model_ids: List[str]):
    """모델별 토론 로그를 생성한다."""
    from experiments.generate import generate_for_model

    for model_id in model_ids:
        logger.info("=" * 60)
        logger.info("생성 시작: %s", model_id)
        result = generate_for_model(model_id)
        logger.info("생성 완료: %s", result)


# ── 2단계: Rule 평가 ────────────────────────────────────────────────────────

def stage_eval_rule():
    """모든 모델의 Rule-based 포맷 평가를 수행한다."""
    from experiments.evaluator_rule import evaluate_model_logs, _save_results

    for model_dir in sorted(LOGS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        logger.info("Rule 평가: %s", model_dir.name)
        results = evaluate_model_logs(model_dir)
        _save_results(model_dir.name, results)


# ── 3단계: LLM 평가 ────────────────────────────────────────────────────────

def stage_eval_llm():
    """모든 모델의 LLM-as-a-Judge 평가를 수행한다."""
    from experiments.evaluator_llm import evaluate_model_logs, _save_results

    for model_dir in sorted(LOGS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        logger.info("LLM 평가: %s", model_dir.name)
        results = evaluate_model_logs(model_dir)
        _save_results(model_dir.name, results)


# ── 4단계: 집계 ─────────────────────────────────────────────────────────────

def _normalize_format_score(score_0_100: float) -> float:
    """0~100점을 1~5 스케일로 선형 변환한다."""
    return 1.0 + (score_0_100 / 100.0) * 4.0


def stage_aggregate() -> Path:
    """Rule + LLM 평가 결과를 합산하여 CSV로 저장한다."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Rule 평가 로드
    rule_scores: Dict[str, Dict[str, float]] = {}  # {model_id: {file: score}}
    rule_dir = EVALS_DIR / "rule"
    if rule_dir.exists():
        for f in rule_dir.glob("*_rule.json"):
            with f.open(encoding="utf-8") as fp:
                data = json.load(fp)
            model_id = f.stem.replace("_rule", "")
            rule_scores[model_id] = {r["file"]: r.get("format_score", 0) for r in data if "file" in r}

    # LLM 평가 로드
    llm_scores: Dict[str, Dict[str, Dict]] = {}  # {model_id: {file: {criteria: score}}}
    llm_dir = EVALS_DIR / "llm"
    if llm_dir.exists():
        for f in llm_dir.glob("*_llm.json"):
            with f.open(encoding="utf-8") as fp:
                data = json.load(fp)
            model_id = f.stem.replace("_llm", "")
            llm_scores[model_id] = {}
            for r in data:
                if "scores" in r:
                    llm_scores[model_id][r["file"]] = r["scores"]

    # Raw CSV 생성
    raw_path = ARTIFACTS_DIR / "results_raw.csv"
    # 토픽→카테고리 매핑
    category_map = {
        "tech": "기술", "econ": "경제", "poli": "정치", "env": "환경",
    }

    fieldnames = [
        "model_id", "file", "topic_id", "category", "preset", "debate_format",
        "format_score_raw", "format_score_norm",
    ] + [f"llm_{c}" for c in LLM_EVAL_CRITERIA] + [
        "llm_mean", "final_score",
    ]

    rows = []
    for model_id in sorted(set(list(rule_scores.keys()) + list(llm_scores.keys()))):
        all_files = set()
        if model_id in rule_scores:
            all_files.update(rule_scores[model_id].keys())
        if model_id in llm_scores:
            all_files.update(llm_scores[model_id].keys())

        for fname in sorted(all_files):
            # Rule score
            fmt_raw = rule_scores.get(model_id, {}).get(fname, 0)
            fmt_norm = _normalize_format_score(fmt_raw)

            # LLM scores
            llm = llm_scores.get(model_id, {}).get(fname, {})
            llm_values = {}
            for c in LLM_EVAL_CRITERIA:
                llm_values[c] = llm.get(c, {}).get("score", 0)

            llm_mean = sum(llm_values.values()) / len(LLM_EVAL_CRITERIA) if llm_values else 0

            # Final score
            # 균등 가중치: Rule 1개 + LLM 8개 = 9개 항목 단순 평균
            all_scores = [fmt_norm] + [llm_values.get(c, 0) for c in LLM_EVAL_CRITERIA]
            final = sum(all_scores) / len(all_scores) if all_scores else 0

            # 파일명에서 메타 추출: tech_001_2v2_balanced.json
            name_no_ext = fname.replace(".json", "")
            parts = name_no_ext.split("_")
            topic_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else ""
            # preset은 토픽ID 이후 전부 (예: 2v2_balanced, 3v3_mixed)
            preset = "_".join(parts[2:]) if len(parts) >= 3 else ""
            cat_prefix = parts[0] if parts else ""
            category = category_map.get(cat_prefix, cat_prefix)
            # 포맷 추출
            fmt = ""
            if "2v2" in preset:
                fmt = "2:2"
            elif "3v3" in preset:
                fmt = "3:3"

            row = {
                "model_id": model_id,
                "file": fname,
                "topic_id": topic_id,
                "category": category,
                "preset": preset,
                "debate_format": fmt,
                "format_score_raw": round(fmt_raw, 1),
                "format_score_norm": round(fmt_norm, 2),
            }
            for c in LLM_EVAL_CRITERIA:
                row[f"llm_{c}"] = llm_values.get(c, 0)
            row["llm_mean"] = round(llm_mean, 2)
            row["final_score"] = round(final, 2)
            rows.append(row)

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Raw CSV 저장: %s (%d행)", raw_path, len(rows))

    # Summary CSV (모델별 평균)
    summary_path = ARTIFACTS_DIR / "results_summary.csv"
    model_agg: Dict[str, List[Dict]] = {}
    for row in rows:
        model_agg.setdefault(row["model_id"], []).append(row)

    summary_rows = []
    for model_id, model_rows in sorted(model_agg.items()):
        n = len(model_rows)
        avg_fmt = sum(r["format_score_raw"] for r in model_rows) / n
        avg_final = sum(r["final_score"] for r in model_rows) / n
        avg_llm = sum(r["llm_mean"] for r in model_rows) / n

        srow = {
            "model_id": model_id,
            "n_experiments": n,
            "avg_format_score": round(avg_fmt, 1),
            "avg_llm_mean": round(avg_llm, 2),
            "avg_final_score": round(avg_final, 2),
        }
        for c in LLM_EVAL_CRITERIA:
            srow[f"avg_{c}"] = round(sum(r[f"llm_{c}"] for r in model_rows) / n, 2)
        summary_rows.append(srow)

    summary_fields = ["model_id", "n_experiments", "avg_format_score", "avg_llm_mean", "avg_final_score"] + \
                     [f"avg_{c}" for c in LLM_EVAL_CRITERIA]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    logger.info("Summary CSV 저장: %s (%d모델)", summary_path, len(summary_rows))

    # Win-rate 테이블
    _compute_win_rate(rows)

    return raw_path


def _compute_win_rate(rows: List[Dict]):
    """Qwen-2.5-32B-Instruct 기준 승무패를 계산한다."""
    hero = "Qwen-2.5-32B-Instruct"
    opponents = [m for m in MODELS if m != hero and m != "GPT-5.4"]

    # (topic_id, stance, format) → {model_id: final_score}
    experiments: Dict[tuple, Dict[str, float]] = {}
    for row in rows:
        key = (row["topic_id"], row["user_stance"], row["debate_format"])
        experiments.setdefault(key, {})[row["model_id"]] = row["final_score"]

    win_rate_rows = []
    for opp in opponents:
        wins, draws, losses = 0, 0, 0
        for key, scores in experiments.items():
            if hero in scores and opp in scores:
                diff = scores[hero] - scores[opp]
                if diff > WIN_THRESHOLD:
                    wins += 1
                elif diff < -WIN_THRESHOLD:
                    losses += 1
                else:
                    draws += 1
        total = wins + draws + losses
        win_rate_rows.append({
            "opponent": opp,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "total": total,
            "win_rate": f"{wins / total * 100:.1f}%" if total > 0 else "N/A",
        })

    wr_path = ARTIFACTS_DIR / "win_rate_table.csv"
    with wr_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["opponent", "wins", "draws", "losses", "total", "win_rate"])
        writer.writeheader()
        writer.writerows(win_rate_rows)

    logger.info("Win-rate 테이블 저장: %s", wr_path)


# ── 5단계: 시각화 ───────────────────────────────────────────────────────────

def stage_report():
    """리포트를 생성한다."""
    from experiments.report import generate_all_charts
    generate_all_charts()


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="A/B 테스트 파이프라인")
    parser.add_argument(
        "--stage",
        choices=["generate", "eval", "eval-rule", "eval-llm", "aggregate", "report", "all"],
        default="all",
    )
    parser.add_argument("--model", type=str, default=None, help="특정 모델만 생성 (generate stage)")
    args = parser.parse_args()

    if args.stage in ("generate", "all"):
        models = [args.model] if args.model else list(MODELS.keys())
        stage_generate(models)

    if args.stage in ("eval", "eval-rule", "all"):
        stage_eval_rule()

    if args.stage in ("eval", "eval-llm", "all"):
        stage_eval_llm()

    if args.stage in ("aggregate", "all"):
        stage_aggregate()

    if args.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
