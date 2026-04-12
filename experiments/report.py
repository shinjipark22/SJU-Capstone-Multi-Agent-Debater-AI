"""
report.py -- 결과 시각화 및 리포팅

결과 CSV를 바탕으로 레이더 차트, 바 차트, 승무패 테이블을 생성한다.
Qwen-2.5-32B-Instruct를 Hero Model로 강조하고,
GPT-5.4를 점선 기준선으로 표시한다.

사용법:
    python -m experiments.report
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from experiments.config import ARTIFACTS_DIR, LLM_EVAL_CRITERIA, MODELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

for _fp in _FONT_PATHS:
    if Path(_fp).exists():
        fm.fontManager.addfont(_fp)
        plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
else:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["axes.unicode_minus"] = False

# ── 모델별 스타일 ───────────────────────────────────────────────────────────
HERO_MODEL = "Qwen-2.5-32B-Instruct"
BASELINE_MODEL = "GPT-5.4"

MODEL_STYLES = {
    "GPT-5.4":                      {"color": "#888888", "linestyle": "--", "linewidth": 1.5, "marker": "o"},
    "Qwen-2.5-32B-Instruct":       {"color": "#FF6B35", "linestyle": "-",  "linewidth": 3.0, "marker": "s"},
    "Gemma-2-27b-it":              {"color": "#4ECDC4", "linestyle": "-",  "linewidth": 1.2, "marker": "^"},
    "EXAONE-3.5-32B-Instruct":    {"color": "#45B7D1", "linestyle": "-",  "linewidth": 1.2, "marker": "D"},
    "DeepSeek-R1-Distill-Qwen-14B": {"color": "#96CEB4", "linestyle": "-",  "linewidth": 1.2, "marker": "v"},
    "Qwen-2.5-7B-Instruct":       {"color": "#DDA0DD", "linestyle": "-",  "linewidth": 1.2, "marker": "p"},
}

# ── 데이터 로드 ─────────────────────────────────────────────────────────────

def _load_summary() -> List[Dict]:
    """results_summary.csv를 로드한다."""
    path = ARTIFACTS_DIR / "results_summary.csv"
    if not path.exists():
        logger.error("results_summary.csv가 없습니다. pipeline aggregate를 먼저 실행하세요.")
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── 9-1. Radar Chart ────────────────────────────────────────────────────────

CRITERIA_LABELS = {
    "self_repetition": "자기반복 방지",
    "team_repetition": "팀반복 방지",
    "role_consistency": "역할 일관성",
    "persona_tone_toxicity": "논조/독성",
    "web_search_tool_use": "검색 활용",
    "faithfulness_hallucination_control": "사실성/할루",
    "logic_evidence_synthesis": "논리/근거",
    "korean_language_compliance": "한국어 준수",
}


def generate_radar_chart(summary: List[Dict]):
    """6개 모델의 7개 LLM 지표를 레이더 차트로 시각화한다."""
    labels = [CRITERIA_LABELS.get(c, c) for c in LLM_EVAL_CRITERIA]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # GPT-5.4를 먼저 (점선 기준선), 나머지 순서대로, Hero를 마지막에 (위에 그려지도록)
    ordered_models = []
    hero_row = None
    baseline_row = None
    for row in summary:
        if row["model_id"] == HERO_MODEL:
            hero_row = row
        elif row["model_id"] == BASELINE_MODEL:
            baseline_row = row
        else:
            ordered_models.append(row)

    if baseline_row:
        ordered_models.insert(0, baseline_row)
    if hero_row:
        ordered_models.append(hero_row)

    for row in ordered_models:
        model_id = row["model_id"]
        values = [float(row.get(f"avg_{c}", 0)) for c in LLM_EVAL_CRITERIA]
        values += values[:1]

        style = MODEL_STYLES.get(model_id, {"color": "gray", "linestyle": "-", "linewidth": 1})
        ax.plot(angles, values, label=model_id, **style)
        if model_id == HERO_MODEL:
            ax.fill(angles, values, alpha=0.15, color=style["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.set_title("LLM Judge: 7개 평가 지표 비교", fontsize=14, fontweight="bold", pad=20)

    out = ARTIFACTS_DIR / "radar_chart.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Radar chart 저장: %s", out)


# ── 9-2. Bar Chart (포맷 준수율) ────────────────────────────────────────────

def generate_format_bar_chart(summary: List[Dict]):
    """모델별 포맷 준수율 평균을 바 차트로 시각화한다."""
    models = [row["model_id"] for row in summary]
    scores = [float(row.get("avg_format_score", 0)) for row in summary]
    colors = [MODEL_STYLES.get(m, {}).get("color", "gray") for m in models]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(models)), scores, color=colors, edgecolor="white", linewidth=0.5)

    # Hero 강조
    for i, m in enumerate(models):
        if m == HERO_MODEL:
            bars[i].set_edgecolor("#FF6B35")
            bars[i].set_linewidth(3)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("포맷 준수율 (0~100)")
    ax.set_ylim(0, 100)
    ax.set_title("모델별 포맷 준수율 비교", fontsize=14, fontweight="bold")

    for i, v in enumerate(scores):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)

    fig.tight_layout()
    out = ARTIFACTS_DIR / "format_bar_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Bar chart 저장: %s", out)


# ── 9-3. Win-rate Table (텍스트 출력) ───────────────────────────────────────

def print_win_rate_table():
    """승무패 테이블을 출력한다."""
    path = ARTIFACTS_DIR / "win_rate_table.csv"
    if not path.exists():
        logger.warning("win_rate_table.csv가 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"  {HERO_MODEL} vs 오픈소스 모델 승무패")
    print(f"{'='*60}")
    print(f"  {'상대 모델':<35} {'승':>4} {'무':>4} {'패':>4} {'승률':>8}")
    print(f"  {'-'*55}")

    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(f"  {row['opponent']:<35} {row['wins']:>4} {row['draws']:>4} {row['losses']:>4} {row['win_rate']:>8}")

    print(f"{'='*60}\n")


# ── 통합 실행 ───────────────────────────────────────────────────────────────

def _load_raw() -> List[Dict]:
    """results_raw.csv를 로드한다."""
    path = ARTIFACTS_DIR / "results_raw.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── 세부 분석: 카테고리별 성능 ──────────────────────────────────────────────

def generate_category_heatmap(raw: List[Dict]):
    """모델 × 카테고리 성능 히트맵."""
    if not raw:
        return

    # 모델×카테고리별 평균 final_score
    data: Dict[str, Dict[str, List[float]]] = {}
    for row in raw:
        model = row["model_id"]
        cat = row.get("category", "")
        score = float(row.get("llm_mean", 0))
        data.setdefault(model, {}).setdefault(cat, []).append(score)

    models = sorted(data.keys())
    categories = sorted(set(cat for m in data.values() for cat in m.keys()))
    if not categories:
        return

    matrix = []
    for model in models:
        row_vals = []
        for cat in categories:
            vals = data.get(model, {}).get(cat, [0])
            row_vals.append(sum(vals) / len(vals))
        matrix.append(row_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=1, vmax=5)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(categories)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if matrix[i][j] > 3.5 else "black")

    fig.colorbar(im, ax=ax, label="LLM Judge 평균 (1~5)")
    ax.set_title("모델 × 카테고리 성능 히트맵", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = ARTIFACTS_DIR / "category_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Category heatmap 저장: %s", out)


# ── 세부 분석: 강경도 프리셋별 성능 ─────────────────────────────────────────

def generate_preset_comparison(raw: List[Dict]):
    """모델 × 강경도 프리셋별 성능 비교 그룹 바 차트."""
    if not raw:
        return

    data: Dict[str, Dict[str, List[float]]] = {}
    for row in raw:
        model = row["model_id"]
        preset = row.get("preset", "")
        score = float(row.get("llm_mean", 0))
        data.setdefault(model, {}).setdefault(preset, []).append(score)

    models = sorted(data.keys())
    presets = sorted(set(p for m in data.values() for p in m.keys()))
    if not presets:
        return

    x = np.arange(len(models))
    width = 0.8 / len(presets)

    fig, ax = plt.subplots(figsize=(14, 6))
    preset_colors = ["#FF6B35", "#4ECDC4", "#45B7D1", "#96CEB4", "#DDA0DD"]

    for i, preset in enumerate(presets):
        vals = [sum(data.get(m, {}).get(preset, [0])) / max(len(data.get(m, {}).get(preset, [1])), 1)
                for m in models]
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=preset, color=preset_colors[i % len(preset_colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LLM Judge 평균 (1~5)")
    ax.set_ylim(0, 5)
    ax.set_title("모델 × 강경도 프리셋별 성능 비교", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out = ARTIFACTS_DIR / "preset_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Preset comparison 저장: %s", out)


# ── 세부 분석: 평가 항목별 모델 비교 ────────────────────────────────────────

def generate_criteria_breakdown(raw: List[Dict]):
    """7개 평가 항목별 모델 성능 그룹 바 차트."""
    if not raw:
        return

    data: Dict[str, Dict[str, List[float]]] = {}
    for row in raw:
        model = row["model_id"]
        for c in LLM_EVAL_CRITERIA:
            score = float(row.get(f"llm_{c}", 0))
            data.setdefault(model, {}).setdefault(c, []).append(score)

    models = sorted(data.keys())
    short_labels = {
        "self_repetition": "자기반복",
        "team_repetition": "팀반복",
        "role_consistency": "역할일관",
        "persona_tone_toxicity": "논조",
        "web_search_tool_use": "검색활용",
        "faithfulness_hallucination_control": "사실성",
        "logic_evidence_synthesis": "논리",
    }

    x = np.arange(len(LLM_EVAL_CRITERIA))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(16, 7))

    for i, model in enumerate(models):
        vals = [sum(data[model].get(c, [0])) / max(len(data[model].get(c, [1])), 1)
                for c in LLM_EVAL_CRITERIA]
        style = MODEL_STYLES.get(model, {})
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=model, color=style.get("color", "gray"))

    ax.set_xticks(x)
    ax.set_xticklabels([short_labels.get(c, c) for c in LLM_EVAL_CRITERIA], fontsize=10)
    ax.set_ylabel("점수 (1~5)")
    ax.set_ylim(0, 5)
    ax.set_title("평가 항목별 모델 성능 비교", fontsize=14, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()

    out = ARTIFACTS_DIR / "criteria_breakdown.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Criteria breakdown 저장: %s", out)


# ── 세부 분석: 포맷별(2:2 vs 3:3) 성능 ──────────────────────────────────────

def generate_format_comparison(raw: List[Dict]):
    """2:2 vs 3:3 포맷별 모델 성능 비교."""
    if not raw:
        return

    data: Dict[str, Dict[str, List[float]]] = {}
    for row in raw:
        model = row["model_id"]
        fmt = row.get("debate_format", "")
        score = float(row.get("llm_mean", 0))
        data.setdefault(model, {}).setdefault(fmt, []).append(score)

    models = sorted(data.keys())
    formats = ["2:2", "3:3"]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, fmt in enumerate(formats):
        vals = [sum(data.get(m, {}).get(fmt, [0])) / max(len(data.get(m, {}).get(fmt, [1])), 1)
                for m in models]
        ax.bar(x + (i - 0.5) * width, vals, width, label=fmt)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LLM Judge 평균 (1~5)")
    ax.set_ylim(0, 5)
    ax.set_title("토론 포맷별 모델 성능 (2:2 vs 3:3)", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()

    out = ARTIFACTS_DIR / "format_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Format comparison 저장: %s", out)


# ── 통합 실행 ───────────────────────────────────────────────────────────────

def generate_all_charts():
    """모든 차트와 테이블을 생성한다."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = _load_summary()
    raw = _load_raw()

    if not summary and not raw:
        logger.error("데이터가 없습니다.")
        return

    # 기본 차트
    if summary:
        generate_radar_chart(summary)
        generate_format_bar_chart(summary)
    print_win_rate_table()

    # 세부 분석
    if raw:
        generate_category_heatmap(raw)
        generate_preset_comparison(raw)
        generate_criteria_breakdown(raw)
        generate_format_comparison(raw)

    logger.info("모든 리포트 생성 완료. 결과: %s", ARTIFACTS_DIR)


def main():
    generate_all_charts()


if __name__ == "__main__":
    main()
