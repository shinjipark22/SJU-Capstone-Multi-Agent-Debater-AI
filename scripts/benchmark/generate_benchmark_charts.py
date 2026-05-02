"""generate_benchmark_charts.py — 벤치마크 루브릭 차트 생성 스크립트.

artifacts/ 에 두 PNG를 생성한다:
  1) baseline_vs_tuned_llm.png — Qwen 2.5-32B 튜닝 전·후 비교
  2) model_comparison_llm.png  — 4모델 LLM 루브릭 비교

디자인 수정 시 이 스크립트만 편집해서 재실행.

사용:
    python scripts/generate_benchmark_charts.py
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 한글 폰트
fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf")
plt.rcParams["font.family"] = "NanumSquareRound"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path("/disk1/SJ/SJU-Capstone-Multi-Agent-Debater-AI")
EVAL_DIR = ROOT / "experiments" / "evals"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 역할 일관성·페르소나 어조는 천장 근처라 제외 (벤치마크 리포트용)
CRITERIA = [
    "self_repetition", "team_repetition",
    "web_search_tool_use", "faithfulness_hallucination_control",
    "logic_evidence_synthesis", "korean_language_compliance",
]
LABELS = ["자기\n반복", "팀\n반복", "웹서치\n활용", "팩트\n충실성", "논리\n종합", "한국어\n준수"]


def averages(model_id: str) -> list:
    """모델별 6지표 평균."""
    data = json.loads((EVAL_DIR / "llm" / f"{model_id}_llm.json").read_text())
    out = []
    for c in CRITERIA:
        vs = [r["scores"].get(c, {}).get("score", 0) for r in data if "scores" in r and r["scores"]]
        vs = [v for v in vs if v]
        out.append(statistics.mean(vs) if vs else 0.0)
    return out


def chart_baseline_vs_tuned():
    baseline = averages("Qwen-2.5-32B-baseline")
    tuned = averages("Qwen-2.5-32B-Instruct")
    x = np.arange(len(CRITERIA))
    w = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    b1 = ax.bar(x - w/2, baseline, w, label="Baseline", color="#A7A29F",
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + w/2, tuned, w, label="프롬프트 tuned", color="#5F6DE2",
                edgecolor="white", linewidth=0.5)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.06, f"{h:.2f}",
                ha="center", fontsize=9, color="#6B6663")
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.06, f"{h:.2f}",
                ha="center", fontsize=9, color="#3545B0", fontweight="bold")
    ax.set_ylabel("LLM Judge 점수 (1~5)", fontsize=11)
    ax.set_title("프롬프트 튜닝 전·후 LLM 루브릭 비교", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylim(0, 6)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.axhline(y=5, color="#D0D0D0", linestyle="--", linewidth=0.8, zorder=0)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    bm, tm = float(np.mean(baseline)), float(np.mean(tuned))
    ax.text(
        0.01, 0.97,
        f"평균\n  Baseline        {bm:.2f}\n  프롬프트 tuned  {tm:.2f}  (+{tm-bm:.2f})",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F3", edgecolor="#D8D8D0"),
    )
    plt.tight_layout()
    out = OUT_DIR / "baseline_vs_tuned_llm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def chart_model_comparison():
    models = [
        ("Qwen-2.5-32B-Instruct", "Qwen 2.5-32B", "#5F6DE2"),
        ("EXAONE-4.0-32B", "EXAONE 4.0-32B", "#E2725B"),
        ("DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Qwen-14B", "#73A942"),
        ("Qwen-2.5-7B-Instruct", "Qwen 2.5-7B", "#C9A961"),
    ]
    scores = {m: averages(m) for m, _, _ in models}
    x = np.arange(len(CRITERIA))
    n = len(models)
    w = 0.8 / n

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    for i, (m, label, color) in enumerate(models):
        offset = (i - (n - 1) / 2) * w
        bars = ax.bar(x + offset, scores[m], w, label=label, color=color,
                      edgecolor="white", linewidth=0.4)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.04, f"{h:.2f}",
                    ha="center", fontsize=7.5, color="#333")
    ax.set_ylabel("LLM Judge 점수 (1~5)", fontsize=11)
    ax.set_title("모델별 LLM 루브릭 점수 비교", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylim(0, 6)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.axhline(y=5, color="#D0D0D0", linestyle="--", linewidth=0.7, zorder=0)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    lines = ["평균"]
    for m, label, _ in models:
        lines.append(f"  {label} : {float(np.mean(scores[m])):.2f}")
    ax.text(
        0.01, 0.98, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F3", edgecolor="#D8D8D0"),
    )
    plt.tight_layout()
    out = OUT_DIR / "model_comparison_llm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    chart_baseline_vs_tuned()
    chart_model_comparison()
